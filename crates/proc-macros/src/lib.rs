use proc_macro2::Ident;
use quote::{format_ident, quote};
use syn::{Attribute, Data, DeriveInput, Fields};

/// Define a save state struct and a method for converting to the save state struct.
///
/// The save state struct will have the name `{struct_name}State`. The original struct will have
/// a new method `save_state(&mut self)` that returns a save state value.
///
/// All fields that are not annotated with a `#[save_state(_)]` attribute must implement `Clone`.
///
/// Struct fields may be annotated with 1 of 2 attributes:
/// - `#[save_state(skip)]`: The field will be left out of the save state struct
/// - `#[save_state(to = OtherState)]`: The field in the save state struct will be of type `OtherState`,
///   and `save_state()` will call the field's `save_state()` method instead of `clone()`
///
/// The save state struct will implement the traits `Debug`, `Clone`, `bincode::Encode`, and
/// `bincode::Decode`.
///
/// No `from_state(state)` method is defined automatically because it is not practical to do so
/// when `#[save_state(to = OtherState)]` is used.
///
/// Example usage:
/// ```
/// use bincode::{Decode, Encode};
/// use proc_macros::SaveState;
///
/// #[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, SaveState)]
/// struct Foo {
///     field: u32,
/// }
///
/// #[derive(SaveState)]
/// struct Bar {
///     a: Foo,
///     #[save_state(to = FooState)]
///     b: Foo,
///     c: i32,
///     #[save_state(skip)]
///     d: i32,
/// }
///
/// let mut bar = Bar {
///     a: Foo { field: 1 },
///     b: Foo { field: 2 },
///     c: 3,
///     d: 4,
/// };
/// let bar_state = bar.save_state();
///
/// assert_eq!(bar_state.a, Foo { field: 1 });
/// assert_eq!(bar_state.b.field, 2);
/// assert_eq!(bar_state.c, 3);
///
/// // Does not compile because d was skipped
/// // assert_eq!(bar_state.d, 4);
/// ```
///
/// # Panics
///
/// This macro only supports structs with named fields, and it will panic if applied to a different
/// data type.
#[proc_macro_derive(SaveState, attributes(save_state))]
pub fn save_state(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input: DeriveInput = syn::parse(input).expect("Failed to parse input");

    let struct_ident = &input.ident;

    let Data::Struct(struct_data) = &input.data else {
        panic!("SaveState macro only supports structs; '{struct_ident}' is not a struct");
    };

    let Fields::Named(fields) = &struct_data.fields else {
        panic!(
            "SaveState macro only supports structs with named fields; '{struct_ident}' does not have named fields"
        );
    };

    let save_state_ident = format_ident!("{struct_ident}State");

    let mut save_state_fields = Vec::new();
    let mut to_state_fields = Vec::new();

    for field in &fields.named {
        let field_ident = field.ident.as_ref().unwrap();
        let field_ty = &field.ty;
        let attribute = field.attrs.iter().find_map(parse_attribute);

        match attribute {
            Some(SaveStateAttribute::Skip) => {}
            Some(SaveStateAttribute::ToState(field_state_ident)) => {
                save_state_fields.push(quote! {
                    #field_ident: #field_state_ident
                });

                to_state_fields.push(quote! {
                    #field_ident: self.#field_ident.save_state()
                });
            }
            None => {
                save_state_fields.push(quote! {
                    #field_ident: #field_ty
                });

                to_state_fields.push(quote! {
                    #field_ident: self.#field_ident.clone()
                });
            }
        }
    }

    let struct_definition = quote! {
        #[derive(Debug, Clone, ::bincode::Encode, ::bincode::Decode)]
        pub struct #save_state_ident {
            #(#save_state_fields,)*
        }
    };

    let to_state_impl = quote! {
        impl #struct_ident {
            #[must_use]
            pub fn save_state(&mut self) -> #save_state_ident {
                #save_state_ident {
                    #(#to_state_fields,)*
                }
            }
        }
    };

    quote! {
        #struct_definition
        #to_state_impl
    }
    .into()
}

#[derive(Clone)]
enum SaveStateAttribute {
    Skip,
    ToState(Ident),
}

fn parse_attribute(attribute: &Attribute) -> Option<SaveStateAttribute> {
    if !attribute.path().is_ident("save_state") {
        return None;
    }

    let mut parsed: Option<SaveStateAttribute> = None;
    attribute
        .parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                parsed = Some(SaveStateAttribute::Skip);
                return Ok(());
            }

            if meta.path.is_ident("to") {
                let value = meta.value()?;
                let field_state_ident: Ident = value.parse()?;
                parsed = Some(SaveStateAttribute::ToState(field_state_ident));
                return Ok(());
            }

            Err(meta.error("Unexpected save_state attribute"))
        })
        .expect("Failed to parse save_state attribute");

    parsed
}
